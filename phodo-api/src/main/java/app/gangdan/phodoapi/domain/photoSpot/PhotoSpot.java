package app.gangdan.phodoapi.domain.photoSpot;

import lombok.*;
import software.amazon.ion.Decimal;

import javax.persistence.*;
import java.math.BigDecimal;

@Entity
@Table(name = "photo_spot")
@Getter
@Builder
@NoArgsConstructor @AllArgsConstructor
public class PhotoSpot {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long photoSpotId;

    private String photoSpotName;

    private BigDecimal longitude;

    private BigDecimal latitude;

}
